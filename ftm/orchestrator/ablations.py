"""The non-LLM planners and the ablation switches that apply to them, on validation (week 9, G4).

Owner: T2 (Bhavya).  Spec: plan.md 7.4 (Table 5b), tracks.md 4.10 week 9 ("all non-LLM planners + ablation configs on val").

Every arm of `configs/experiments/ablations_nonllm.yaml` is run on the same alerts through the same harness, gate and
accountant; only the settings it lists differ from the base. The validation alerts are cut in time (`split_val`): the `fit`
half trains what needs training (the learned policy, the fitted fusions), every arm is scored on the `eval` half.

Per arm: Brier before and after with a cluster-bootstrap CI of the gain (T4's `ftm.eval.bootstrap`, clusters = alerts that
share an account), the paired Brier difference to the reference arm with its CI, questions and eps per alert, the share of
questions that got no reply (declines, timeouts, refusals) and simulated days per alert. Arms whose inputs are missing
(no enumerated table, no exact-label table, no LightGBM) come back as `skipped` with the reason, never silently dropped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml

from ftm.backtest.engine import BacktestConfig, BtAlert
from ftm.banks.partner import BankConfig
from ftm.eval import bootstrap as bs
from ftm.eval.g2 import answer_log_lrs, clusters_for
from ftm.fusion.fuse import Fusion, answer_features
from ftm.orchestrator.eps_sweep import fixed_level
from ftm.orchestrator.policy_data import collect_policy_data
from ftm.orchestrator.runner import build_planner, make_harness, pool_from_backtest, run_alerts
from ftm.planners.learned import LearnedPolicy, split_val
from ftm.planners.pool import QuestionPool
from ftm.privacy.accountant import PrivacyAccountant
from ftm.types import Limits, Reply

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "ablations_nonllm.yaml"
KEYS = ("budget", "cap", "one_per_family", "waves", "threshold", "p_decline", "p_timeout", "fusion", "pool")


@dataclass(frozen=True)
class Arm:
    name: str
    group: str
    system: str
    settings: dict = field(default_factory=dict)  # `base` with this arm's overrides

    @property
    def pool_name(self) -> str:
        return self.settings["pool"]


@dataclass
class AblationSpec:
    arms: list[Arm]
    reference: str
    eps_levels: tuple[float, ...]
    fit_frac: float


def load_spec(path: str | Path = CONFIG) -> AblationSpec:
    c = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    base = c["base"]
    unknown = set(base) - set(KEYS)
    if unknown:
        raise ValueError(f"unknown base settings {sorted(unknown)}; known: {KEYS}")
    arms = []
    for a in c["arms"]:
        extra = set(a) - {"name", "group", "system", *KEYS}
        if extra:
            raise ValueError(f"arm {a.get('name')!r}: unknown settings {sorted(extra)}; known: {KEYS}")
        s = {**base, **{k: a[k] for k in KEYS if k in a}}
        if a["system"] == "enumerator" and "pool" not in a:
            s["pool"] = "enumerator"
        arms.append(Arm(a["name"], a["group"], a["system"], s))
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique")
    if c["reference"] not in names:
        raise ValueError(f"reference arm {c['reference']!r} is not one of the arms")
    return AblationSpec(arms, c["reference"], tuple(c["eps_levels"]), float(c["fit_frac"]))


def _banks_for(banks: dict, s: dict) -> dict:
    """The partner banks under this arm's refusal / timeout rates (the same ledgers and seeds)."""
    if s["p_decline"] is None and s["p_timeout"] is None:
        return banks
    out = {}
    for b, bank in banks.items():
        cfg: BankConfig = bank.cfg
        cfg = replace(cfg, p_decline=cfg.p_decline if s["p_decline"] is None else s["p_decline"],
                      p_timeout=cfg.p_timeout if s["p_timeout"] is None else s["p_timeout"])  # fmt: skip
        out[b] = bank._with_config(cfg)
    return out


def fit_fusion(mode: str, banks: dict, fit: Sequence[BtAlert], pool: QuestionPool, eps_levels: Sequence[float],
               seed: int = 0, limits: Limits | None = None) -> Fusion:  # fmt: skip
    """T4's fusion, fitted on the `fit` alerts: run the summed (naive) EIG planner there, then fit on the per-family sums of
    the back-tested log-LRs of the answers it collected."""
    families = sorted({c.family for c in pool})
    base = {"run_id": "fusion-fit", "commit": "", "machine": "", "system": "eig", "variant": "fusion_fit", "seed": seed}
    res = run_alerts(build_planner("eig", pool, seed), fit, make_harness(banks, {a.alert_id: a.binding for a in fit},
                     eps_levels=eps_levels), limits or Limits(), base, "val")  # fmt: skip
    ids, fams, llr = answer_log_lrs(res.verdicts, pool)
    X, _ = answer_features([a.alert_id for a in fit], ids, fams, llr, families=families)
    p0 = np.array([a.p0 for a in fit])
    y = np.array([float(a.label) for a in fit])
    return Fusion(mode, families=families).fit(p0, X, y)


def _refused_share(verdicts) -> float:
    asked = [e for v in verdicts for e in v.questions_log]
    if not asked:
        return 0.0
    return sum(not isinstance(e.response, Reply) for e in asked) / len(asked)


@dataclass
class PreparedArm:
    """Everything one arm needs to run on a set of alerts: the planner, the harness (gate, accountant, banks under this arm's
    refusal rates) and the limits. Built by `prepare_arm`, the one place the arm settings turn into objects."""

    planner: object
    harness: object
    limits: Limits
    pool: QuestionPool


def prepare_arm(arm: Arm, spec: AblationSpec, *, banks: dict, bindings: dict, fit: Sequence[BtAlert], fit_banks: dict | None,
                pools: dict[str, QuestionPool], policy: LearnedPolicy | None, seed: int, cache: dict) -> PreparedArm | str:  # fmt: skip
    """Build one arm, or return the reason it cannot run (missing table, no LightGBM, ...).

    `banks` and `bindings` are the world the arm is scored in (validation at G4, the test alerts at G5). `fit` and `fit_banks`
    are where whatever needs fitting is fitted (the fitted fusions, the learned policy): the first half of validation, in both.
    `cache` shares those fits between arms."""
    s = arm.settings
    pool = pools.get(s["pool"])
    if pool is None or not len(pool):
        return f"no {s['pool']!r} question table"
    fusion = None
    if s["fusion"]:
        key = ("fusion", s["fusion"], s["pool"])
        if key not in cache:
            cache[key] = fit_fusion(s["fusion"], fit_banks if fit_banks is not None else banks, fit, pool, spec.eps_levels, seed)
        fusion = cache[key]
    pol = None
    if arm.system == "learned":
        if policy is None and "policy" not in cache:
            try:
                data = collect_policy_data(fit_banks if fit_banks is not None else banks, fit, pool,
                                           eps_per_alert=s["budget"], seed=seed)  # fmt: skip
                cache["policy"] = LearnedPolicy.fit(data, seed=seed)
            except ImportError as e:
                cache["policy"] = None
                cache["policy_error"] = f"lightgbm is not installed ({e})"
            except ValueError as e:
                cache["policy"] = None
                cache["policy_error"] = str(e)
        pol = policy or cache.get("policy")
        if pol is None:
            return cache.get("policy_error", "no fitted policy")
    acc = PrivacyAccountant(eps_per_alert=s["budget"], per_account_cap=s["cap"])
    harness = make_harness(_banks_for(banks, s), bindings, acc, eps_levels=spec.eps_levels)
    waves = tuple(s["waves"]) if s["waves"] else None
    planner = build_planner(arm.system, pool, seed, eps=fixed_level(s["budget"], spec.eps_levels), threshold=s["threshold"],
                            waves=waves, policy=pol, fusion=fusion, one_per_family=s["one_per_family"])  # fmt: skip
    return PreparedArm(planner, harness, Limits(eps_per_alert=s["budget"]), pool)


def run_ablations(banks: dict, val: Sequence[BtAlert], pools: dict[str, QuestionPool], spec: AblationSpec | None = None,
                  policy: LearnedPolicy | None = None, seed: int = 0, n_boot: int = 500,
                  only: Sequence[str] | None = None) -> list[dict]:  # fmt: skip
    """One row per arm. `val` are the validation alerts (cut here into fit and eval); `pools` maps pool names
    (`realistic`, and optionally `exact_labels`, `enumerator`) to back-tested tables."""
    spec = spec or load_spec()
    fit, ev = split_val(val, spec.fit_frac)
    if not ev:
        raise ValueError("no alerts left to score: lower fit_frac or give more validation alerts")
    bindings = {a.alert_id: a.binding for a in ev}
    y = np.array([float(a.label) for a in ev])
    p0 = np.array([a.p0 for a in ev])
    clusters = clusters_for(ev)
    rows: list[dict] = []
    ps: dict[str, np.ndarray] = {}
    cache: dict = {}
    for arm in spec.arms:
        if only and arm.name not in only and arm.name != spec.reference:
            continue
        s = arm.settings
        prep = prepare_arm(arm, spec, banks=banks, bindings=bindings, fit=fit, fit_banks=banks, pools=pools, policy=policy,
                           seed=seed, cache=cache)  # fmt: skip
        if isinstance(prep, str):
            rows.append(_skipped(arm, prep))
            continue
        base = {"run_id": f"ablation-{arm.name}", "commit": "", "machine": "", "system": arm.system, "variant": arm.name,
                "seed": seed}  # fmt: skip
        res = run_alerts(prep.planner, ev, prep.harness, prep.limits, base, "val")
        p = np.array([a.p0 if v.probability is None else v.probability for a, v in zip(ev, res.verdicts, strict=True)])
        ps[arm.name] = p
        gain = (p0 - y) ** 2 - (p - y) ** 2
        lo, hi = bs.percentile_ci(bs.bootstrap_means(gain, clusters, n_boot, seed).ravel())
        rows.append({"arm": arm.name, "group": arm.group, "system": arm.system, "settings": _jsonable(s),
                     "alerts": len(ev), "positives": int(y.sum()), "brier_p0": res.summary["brier_p0"],
                     "brier_final": res.summary["brier_final"], "brier_gain": float(gain.mean()),
                     "brier_gain_ci": [lo, hi], "mean_questions": res.summary["mean_questions"],
                     "mean_eps": res.summary["mean_eps"], "refused_share": _refused_share(res.verdicts),
                     "mean_sim_days": res.summary["mean_sim_days"], "p95_sim_days": res.summary["p95_sim_days"],
                     "mean_seconds_per_alert": res.summary["mean_seconds_per_alert"], "skipped": None})  # fmt: skip
    ref = ps.get(spec.reference)
    for r in rows:
        if r["skipped"] is None and ref is not None:
            if r["arm"] == spec.reference:
                r["brier_diff_vs_ref"], r["brier_diff_ci"] = 0.0, [0.0, 0.0]
            else:
                d = bs.brier_diff(y, ps[r["arm"]], ref, clusters, n_boot, seed)
                r["brier_diff_vs_ref"], r["brier_diff_ci"] = d["diff"], list(d["ci"])
    return rows


def _skipped(arm: Arm, why: str) -> dict:
    return {"arm": arm.name, "group": arm.group, "system": arm.system, "settings": _jsonable(arm.settings), "skipped": why}


def _jsonable(s: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()}


def format_table(rows: Sequence[dict], reference: str = "eig") -> str:
    """The G4 table as markdown, one row per arm."""
    head = ("| Group | Arm | Brier p0 | Brier final | Gain [95% CI] | vs " + reference + " [95% CI] | Qs | eps | Refused | Days |\n"
            "|---|---|---|---|---|---|---|---|---|---|")  # fmt: skip
    out = [head]
    for r in rows:
        if r.get("skipped"):
            out.append(f"| {r['group']} | {r['arm']} | skipped: {r['skipped']} | | | | | | | |")
            continue
        lo, hi = r["brier_gain_ci"]
        dlo, dhi = r.get("brier_diff_ci", (math.nan, math.nan))
        out.append(f"| {r['group']} | {r['arm']} | {r['brier_p0']:.4f} | {r['brier_final']:.4f} | {r['brier_gain']:+.4f} "
                   f"[{lo:+.4f}, {hi:+.4f}] | {r.get('brier_diff_vs_ref', math.nan):+.4f} [{dlo:+.4f}, {dhi:+.4f}] | "
                   f"{r['mean_questions']:.2f} | {r['mean_eps']:.2f} | {r['refused_share']:.1%} | {r['mean_sim_days']:.2f} |")  # fmt: skip
    return "\n".join(out)


@dataclass
class SyntheticEnv:
    banks: dict
    val: list[BtAlert]
    pools: dict[str, QuestionPool]


def synthetic_env(n_alerts: int = 600, seed: int = 0, eps_levels: Sequence[float] = (0.5, 1.0, 2.0), max_programs: int = 14,
                  signal: float = 1.0, enumerator: bool = True) -> SyntheticEnv:  # fmt: skip
    """The synthetic world for smoke tests, never a result: back-test on the first 50% of the alerts in time (the expert
    questions, and a thinned enumeration for the enumerator), the rest play the validation set with p0 = the base rate."""
    from ftm.backtest.synth import make_world
    from ftm.dsl.enumerate import enumerate_programs
    from ftm.dsl.expert import expert_programs
    from ftm.dsl.parser import canonical

    w = make_world(n_alerts, seed, signal)
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    cut = len(alerts) // 2
    train = alerts[:cut]
    rate = sum(a.label for a in train) / len(train)
    val = [replace(a, p0=rate, split="val") for a in alerts[cut:]]
    cfg = BacktestConfig(min_answers=50, min_per_class=8, min_per_bin=12)
    allp = [p["program"] for p in expert_programs("receiver")]
    progs = allp[:: max(1, len(allp) // max_programs)]
    pool, _ = pool_from_backtest(w.banks, train, progs, eps_levels, cfg)
    pools = {"realistic": pool}
    if enumerator:  # the thinned enumeration costs as much again; only the enumerator arm reads it
        enum_progs = [canonical(n) for n in enumerate_programs(3, ("receiver",))][::6]
        pools["enumerator"], _ = pool_from_backtest(w.banks, train, enum_progs, eps_levels, cfg)
    return SyntheticEnv(w.banks, val, pools)


def synthetic_ablations(n_alerts: int = 600, seed: int = 0, spec: AblationSpec | None = None, n_boot: int = 200,
                        only: Sequence[str] | None = None, max_programs: int = 14, signal: float = 1.0) -> list[dict]:  # fmt: skip
    """The arms end to end on the synthetic world (smoke test, never a result)."""
    spec = spec or load_spec()
    env = synthetic_env(n_alerts, seed, spec.eps_levels[1:], max_programs, signal)  # 0.5, 1, 2 as in the other synthetic checks
    return run_ablations(env.banks, env.val, env.pools, spec, seed=seed, n_boot=n_boot, only=only)

