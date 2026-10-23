"""Gate G2: expert questions help at eps = 2 on validation, and the privacy code stays under its ceiling.

Owner: T4 (Shreyas).  Spec: plan.md 13 (gates), tracks.md 2.2.  Script: `scripts/g2_check.py`.

G2 asks two things of the pieces built so far: (1) the expert question list, asked in the fixed expert order inside a
per-alert budget of eps = 2 through the real harness (gate, accountant, clock), moves the validation Brier score down by
at least 0.01; (2) no back-test LR is above the privacy ceiling beyond sampling error (`ftm.eval.ceiling.backtest_check`).

The run uses T2's code end to end and re-implements none of it: the back-test (`Backtester`, `accept`), the pool of
accepted questions (`QuestionPool`), the planner (`FixedOrderPlanner` via `build_planner`) and the runner (`run_alerts`).
What is new here is the scoring:

  p0            the one-bank score of each alert
  posterior     p0 plus the alert's answers through fusion: summed log LR per family -> `Fusion.predict`. Naive fusion
                (a = 1, w = 1) by default. `per_family` fusion is CROSS-FITTED inside the validation alerts: each fold
                of whole clusters is scored by a model fitted on the other folds, so no alert is scored by a model
                that saw it (the pattern of `fuse.shuffled_answers_test`)
  gain          Brier(p0) - Brier(posterior), point estimate, with a paired cluster-bootstrap CI
  clusters      attempt_id when the alert set has it, plus shared accounts (sender, receiver): the plan's definition
                (`ftm.eval.bootstrap.cluster_ids`); one cluster per alert when neither links anything

PASS iff gain >= GAIN_MIN (the point estimate; the CI is reported next to it) and the ceiling check has no violations.
A pool with no accepted question is a FAIL (gain 0), not a missing run. Nothing here reads a label before the planner
has finished an alert (labels go to the scoring only), and nothing here touches the test split.

Result fields (`run_g2` returns this dict; the script writes it as JSON):
  world, fusion, ask_eps, budget, n_alerts, n_pos, pool, brier_p0, brier_post, gain, gain_ci95, gain_naive,
  gain_ok, clusters, mean_questions, mean_eps, max_eps, budget_ok, families_used, fusion_matches_planner,
  ceiling {cells_judged, entries_skipped, violations, ok}, ceiling_ok, passed.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from ftm.backtest.engine import (
    BacktestCache,
    BacktestConfig,
    Backtester,
    BtAlert,
    accept,
    lr_table_rows,
)
from ftm.dsl.expert import expert_programs
from ftm.dsl.parser import canonical, parse
from ftm.eval import bootstrap as bs
from ftm.eval import ceiling
from ftm.fusion.fuse import Fusion, answer_features
from ftm.orchestrator.runner import build_planner, make_harness, run_alerts
from ftm.planners.pool import QuestionPool
from ftm.privacy.accountant import PrivacyAccountant
from ftm.types import Limits, Reply

GAIN_MIN = 0.01  # G2: at least this much Brier improvement at eps = 2 on validation
BUDGET = 2.0  # eps per alert
ASK_EPS = 0.5  # eps level of each expert question (four of them fit in the budget)
EPS_LEVELS = (0.5, 1.0, 2.0)  # back-tested levels (scripts/backtest.py default)
SYNTH_CONFIG = BacktestConfig(min_answers=50, min_per_class=8, min_per_bin=12)  # small world: looser eligibility
FUSIONS = ("naive", "per_family")
SYNTH_PROGRAMS = (  # expert questions that carry signal in the synthetic world (ftm.backtest.synth)
    "n_counterparties(ALERT.receiver, dir=in, window=before(72))",
    "n_tx(ALERT.receiver, dir=in, window=around(72), max_amount=2000)",
    "passed_on(ALERT.receiver, window=after(24), min_frac=0.5)",
    "returns_home(ALERT.receiver, window=after(72))",
    "dormant(ALERT.receiver, days=7)",
)


def expert_program_list(slots: Sequence[str] = ("receiver", "sender")) -> list[str]:
    """The expert questions (both subject slots), deduplicated by canonical form, as `scripts/backtest.py` builds them."""
    progs = [p["program"] for s in slots for p in expert_programs(s)]
    return list(dict.fromkeys(canonical(parse(p)) for p in progs))


# --- the pool and the cache entries -------------------------------------------------------------------------------
def backtest_entries(banks, train: Sequence[BtAlert], programs: Sequence[str], eps_levels: Sequence[float] = EPS_LEVELS,
                     config: BacktestConfig | None = None, cache: BacktestCache | None = None,
                     fdr: float = 0.10) -> list[dict]:  # fmt: skip
    """Back-test `programs` on the training alerts (a hit in `cache` is reused) and accept with e-BH."""
    bt = Backtester(banks, train, config or BacktestConfig(), cache=cache)
    return accept(bt.run_batch(programs, eps_levels), fdr)


def entries_from_cache(cache: BacktestCache, fdr: float = 0.10) -> list[dict]:
    """The entries of an existing cache directory (every shard of `scripts/backtest.py`), e-BH applied."""
    return accept(cache.entries(), fdr)


# --- answers through fusion ---------------------------------------------------------------------------------------
def answer_log_lrs(verdicts, pool: QuestionPool) -> tuple[list[str], list[str], list[float]]:
    """(alert ids, families, log LRs) of every answered question with a back-tested LR, from the harness's own log.

    The same numbers `PlanSession` adds to its posterior: the pool candidate of the (program, eps), its LR for the answer cell.
    A decline, a timeout, a refusal or an answer cell with no LR contributes nothing."""
    by_question = {(c.program, c.eps): c for c in pool}  # one program has a different LR table at each eps
    ids, fams, llr = [], [], []
    for v in verdicts:
        for e in v.questions_log:
            c = by_question.get((e.program, e.eps))
            if c is None or not isinstance(e.response, Reply):
                continue
            x = c.log_lr(e.response.value)
            if x is not None:
                ids.append(v.alert_id)
                fams.append(c.family)
                llr.append(x)
    return ids, fams, llr


def fold_labels(clusters: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """A fold per alert, with every cluster whole in one fold (related alerts are never split between fit and score)."""
    uniq = np.unique(clusters)
    k = max(1, min(n_folds, len(uniq)))
    fold_of = dict(zip(np.random.default_rng(seed).permutation(uniq), np.arange(len(uniq)) % k, strict=True))
    return np.array([fold_of[c] for c in clusters])


def crossfit_posterior(p0, X, y, clusters, mode: str = "per_family", n_folds: int = 5, seed: int = 0,
                       ridge: float = 1.0) -> np.ndarray:  # fmt: skip
    """Posterior of every alert from a fusion model fitted on the other folds only. `naive` has nothing to fit."""
    p0, y = np.asarray(p0, "f8"), np.asarray(y, "f8")
    if mode == "naive" or X.shape[1] == 0:
        return Fusion("naive").predict(p0, X) if X.shape[1] else p0.copy()
    folds = fold_labels(np.asarray(clusters), n_folds, seed)
    if len(np.unique(folds)) < 2:
        raise ValueError("cross-fitting needs at least two clusters")
    out = np.empty(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        out[te] = Fusion(mode, ridge=ridge).fit(p0[tr], X[tr], y[tr]).predict(p0[te], X[te])
    return out


# --- the check ----------------------------------------------------------------------------------------------------
def clusters_for(alerts: Sequence[BtAlert], attempt_of: dict | None = None, hub_cap: int | None = None) -> np.ndarray:
    """Cluster label per alert: same attempt (when known) or a shared account (`bootstrap.cluster_ids`)."""
    att = [(attempt_of or {}).get(a.alert_id) for a in alerts]
    return bs.cluster_ids(att, [[a.binding.sender for a in alerts], [a.binding.receiver for a in alerts]], hub_cap)


def run_g2(banks, entries: list[dict], val: Sequence[BtAlert], world: str = "synth", fusion: str = "naive",
           budget: float = BUDGET, ask_eps: float = ASK_EPS, eps_levels: Sequence[float] = EPS_LEVELS,
           attempt_of: dict | None = None, hub_cap: int | None = None, n_boot: int = bs.N_BOOT, seed: int = 0,
           n_folds: int = 5, gain_min: float = GAIN_MIN, fdr: float = 0.10) -> dict:  # fmt: skip
    """The G2 check. `entries` are the back-test cache entries (e-BH is applied here, over all of them), `val` the
    validation alerts, `banks` the partner banks the validation alerts are put to."""
    if fusion not in FUSIONS:
        raise ValueError(f"fusion must be one of {FUSIONS}")
    val = list(val)
    if not val:
        raise ValueError("no validation alerts")
    entries = accept([dict(e) for e in entries], fdr)  # e-BH over the whole batch (idempotent if already applied)
    pool = QuestionPool.from_rows(lr_table_rows(entries))
    acc = PrivacyAccountant(eps_per_alert=budget)
    harness = make_harness(banks, {a.alert_id: a.binding for a in val}, accountant=acc, eps_levels=eps_levels)
    planner = build_planner("fixed", pool, seed, eps=ask_eps)  # the expert order, one eps level, no stop threshold
    base = {"run_id": uuid.uuid4().hex[:8], "commit": "", "machine": "t4", "system": "fixed", "variant": "g2",
            "seed": seed}  # fmt: skip
    res = run_alerts(planner, val, harness, Limits(eps_per_alert=budget), base)

    ids = [a.alert_id for a in val]
    y = np.array([float(a.label) for a in val])
    p0 = np.array([a.p0 for a in val])
    a_ids, a_fams, a_llr = answer_log_lrs(res.verdicts, pool)
    fams = sorted(set(a_fams))
    X, _ = answer_features(ids, a_ids, a_fams, a_llr, families=fams) if fams else (np.zeros((len(ids), 0)), [])
    cl = clusters_for(val, attempt_of, hub_cap)
    naive = crossfit_posterior(p0, X, y, cl, "naive")
    post = naive if fusion == "naive" else crossfit_posterior(p0, X, y, cl, fusion, n_folds, seed)
    d = bs.brier_diff(y, p0, post, cl, n_boot=n_boot, seed=seed)  # Brier(p0) - Brier(post): positive = the answers help
    gain = d["diff"]
    gain_naive = float(np.mean((p0 - y) ** 2) - np.mean((naive - y) ** 2))

    spent = [sum(e.response.eps_spent for e in v.questions_log if isinstance(e.response, Reply)) for v in res.verdicts]
    n_q = [sum(isinstance(e.response, Reply) for e in v.questions_log) for v in res.verdicts]
    ceil = ceiling.backtest_check(entries)
    gain_ok = bool(gain >= gain_min)
    return {
        "world": world, "fusion": fusion, "ask_eps": ask_eps, "budget": budget, "n_alerts": len(val),
        "n_pos": int(y.sum()), "pool": len(pool), "brier_p0": float(np.mean((p0 - y) ** 2)),
        "brier_post": float(np.mean((post - y) ** 2)), "gain": float(gain), "gain_ci95": [float(x) for x in d["ci"]],
        "gain_naive": gain_naive, "gain_min": gain_min, "gain_ok": gain_ok, "clusters": bs.cluster_report(cl),
        "mean_questions": float(np.mean(n_q)), "mean_eps": float(np.mean(spent)), "max_eps": float(np.max(spent)),
        "budget_ok": bool(np.max(spent) <= budget + 1e-9 and all(
            e.gate_decision == "allowed" for v in res.verdicts for e in v.questions_log)),
        "families_used": fams,
        # naive fusion by hand must equal the planner's own running posterior: a check that X is built right
        "fusion_matches_planner": bool(np.allclose(naive, [v.probability for v in res.verdicts], atol=1e-6)),
        "ceiling": ceil, "ceiling_ok": bool(ceil["ok"]), "passed": bool(gain_ok and ceil["ok"]),
    }  # fmt: skip


def verdict_line(r: dict) -> str:
    lo, hi = r["gain_ci95"]
    return (f"G2 {'PASS' if r['passed'] else 'FAIL'} [{r['world']}, {r['fusion']} fusion, eps {r['budget']:g}]: "
            f"Brier {r['brier_p0']:.4f} -> {r['brier_post']:.4f}, gain {r['gain']:+.4f} "
            f"(95% CI {lo:+.4f} to {hi:+.4f}; need >= {r['gain_min']:g}) {'ok' if r['gain_ok'] else 'NOT MET'}; "
            f"ceiling {r['ceiling']['cells_judged']} cells judged, {len(r['ceiling']['violations'])} violations "
            f"{'ok' if r['ceiling_ok'] else 'NOT MET'}; {r['n_alerts']} alerts ({r['n_pos']} pos), "
            f"{r['mean_questions']:.1f} questions/alert, pool {r['pool']}"
            + ("" if r["budget_ok"] else "; WARNING: a budget or gate breach in the run"))  # fmt: skip


# --- the synthetic world (tests and smoke runs; never a result) ---------------------------------------------------
def synthetic_inputs(n_alerts: int = 360, seed: int = 0, signal: float = 1.0, programs: Sequence[str] | None = None,
                     train_frac: float = 0.6, eps_levels: Sequence[float] = EPS_LEVELS) -> tuple[dict, list[dict], list[BtAlert]]:  # fmt: skip
    """(banks, back-test entries, validation alerts) of the synthetic world, built the way `synthetic_experiment` does:
    back-test on the first `train_frac` in time, p0 = the training base rate, validate on the rest."""
    from ftm.backtest.synth import make_world

    w = make_world(n_alerts, seed, signal)
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    cut = int(len(alerts) * train_frac)
    train, val = alerts[:cut], alerts[cut:]
    rate = sum(a.label for a in train) / len(train)
    val = [replace(a, p0=rate, split="val") for a in val]
    return w.banks, backtest_entries(w.banks, train, programs or SYNTH_PROGRAMS, eps_levels, SYNTH_CONFIG), val

